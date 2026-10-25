use std::{
    fs,
    path::{Component, Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum CpuSource {
    CgroupV1,
    CgroupV2,
    Host,
}

impl CpuSource {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::CgroupV1 => "cgroup_v1",
            Self::CgroupV2 => "cgroup_v2",
            Self::Host => "host_fallback",
        }
    }
}

#[derive(Clone, Debug)]
struct Controller {
    directory: PathBuf,
    mount: PathBuf,
}

impl Controller {
    fn discover(membership: &str, mounts: &str, controller: Option<&str>) -> Option<Self> {
        let group = membership.lines().find_map(|line| {
            let mut fields = line.splitn(3, ':');
            fields.next()?;
            let controllers = fields.next()?;
            let path = fields.next()?;
            let matches = match controller {
                Some(name) => controllers.split(',').any(|value| value == name),
                None => controllers.is_empty(),
            };
            matches.then_some(path)
        })?;
        mounts.lines().find_map(|line| {
            let (before, after) = line.split_once(" - ")?;
            let mut mount_fields = before.split_whitespace();
            let mut fs_fields = after.split_whitespace();
            let kind = fs_fields.next()?;
            fs_fields.next()?;
            let options = fs_fields.next()?;
            let matches_controller = match controller {
                Some(name) => kind == "cgroup" && options.split(',').any(|value| value == name),
                None => kind == "cgroup2",
            };
            if !matches_controller {
                return None;
            }
            let root = PathBuf::from(unescape_mount_path(mount_fields.nth(3)?));
            let mount = PathBuf::from(unescape_mount_path(mount_fields.next()?));
            let group = Path::new(group);
            // A cgroup namespace can expose membership relative to its own root,
            // while mountinfo still names the root in the host hierarchy.
            let relative = group
                .strip_prefix(&root)
                .or_else(|_| group.strip_prefix("/"))
                .ok()?;
            if relative
                .components()
                .any(|component| matches!(component, Component::ParentDir))
            {
                return None;
            }
            let directory = mount.join(relative);
            directory
                .is_dir()
                .then_some(Controller { directory, mount })
        })
    }

    fn ancestors(&self) -> impl Iterator<Item = &Path> {
        self.directory
            .ancestors()
            .take_while(|path| path.starts_with(&self.mount))
    }
}

#[derive(Debug)]
enum CpuFiles {
    V1 {
        cpu: Controller,
        cpuacct: Controller,
        cpuset: Option<Controller>,
    },
    V2(Controller),
}

impl CpuFiles {
    fn discover(membership: &str, mounts: &str) -> Option<Self> {
        if let Some(cpu) = Controller::discover(membership, mounts, None)
            && cpu.directory.join("cpu.stat").is_file()
            && cpu.directory.join("cpu.max").is_file()
        {
            return Some(Self::V2(cpu));
        }
        Some(Self::V1 {
            cpu: Controller::discover(membership, mounts, Some("cpu"))?,
            cpuacct: Controller::discover(membership, mounts, Some("cpuacct"))?,
            cpuset: Controller::discover(membership, mounts, Some("cpuset")),
        })
    }

    fn source(&self) -> CpuSource {
        match self {
            Self::V1 { .. } => CpuSource::CgroupV1,
            Self::V2(_) => CpuSource::CgroupV2,
        }
    }

    fn read_usage_ns(&self) -> Option<u64> {
        match self {
            Self::V1 { cpuacct, .. } => fs::read_to_string(cpuacct.directory.join("cpuacct.usage"))
                .ok()?
                .trim()
                .parse()
                .ok(),
            Self::V2(cpu) => {
                let stat = fs::read_to_string(cpu.directory.join("cpu.stat")).ok()?;
                let usage_us = stat.lines().find_map(|line| {
                    let mut fields = line.split_whitespace();
                    match (fields.next(), fields.next()) {
                        (Some("usage_usec"), Some(value)) => value.parse::<u64>().ok(),
                        _ => None,
                    }
                })?;
                usage_us.checked_mul(1_000)
            }
        }
    }

    fn read(&self) -> Option<(u64, f64)> {
        let usage_ns = self.read_usage_ns()?;
        let (cpu, cpuset) = match self {
            Self::V1 { cpu, cpuset, .. } => (cpu, cpuset.as_ref()),
            Self::V2(cpu) => (cpu, Some(cpu)),
        };
        let mut capacity_cores: Option<f64> = None;
        for path in cpu.ancestors() {
            let (quota, period) = match self {
                Self::V1 { .. } => (
                    fs::read_to_string(path.join("cpu.cfs_quota_us")).ok()?,
                    fs::read_to_string(path.join("cpu.cfs_period_us")).ok()?,
                ),
                Self::V2(_) => {
                    let limit = match fs::read_to_string(path.join("cpu.max")) {
                        Ok(limit) => limit,
                        // The physical v2 root has no CPU bandwidth interface.
                        Err(error)
                            if path == cpu.mount
                                && error.kind() == std::io::ErrorKind::NotFound =>
                        {
                            continue;
                        }
                        Err(_) => return None,
                    };
                    let mut fields = limit.split_whitespace();
                    (fields.next()?.to_owned(), fields.next()?.to_owned())
                }
            };
            let period: u64 = period.trim().parse().ok()?;
            if period == 0 {
                return None;
            }
            let quota = quota.trim();
            if quota == "max" || (matches!(self, Self::V1 { .. }) && quota == "-1") {
                continue;
            }
            let quota: u64 = quota.parse().ok()?;
            if quota == 0 {
                return None;
            }
            let cores = quota as f64 / period as f64;
            capacity_cores = Some(capacity_cores.map_or(cores, |current| current.min(cores)));
        }
        // Preserve the host fallback for an unlimited hierarchy. A finite quota
        // may still be further constrained by a cpuset inherited from a parent.
        let mut capacity_cores = capacity_cores?;
        if let Some(cpuset) = cpuset {
            for path in cpuset.ancestors() {
                for name in [
                    "cpuset.cpus.effective",
                    "cpuset.effective_cpus",
                    "cpuset.cpus",
                ] {
                    if let Ok(cpus) = fs::read_to_string(path.join(name))
                        && !cpus.trim().is_empty()
                    {
                        capacity_cores = capacity_cores.min(cpu_count(&cpus)? as f64);
                        break;
                    }
                }
            }
        }
        Some((usage_ns, capacity_cores))
    }
}

fn unescape_mount_path(path: &str) -> String {
    path.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

fn cpu_count(cpus: &str) -> Option<u64> {
    cpus.trim()
        .split(',')
        .try_fold(0u64, |count, range| {
            let (start, end) = range.split_once('-').unwrap_or((range, range));
            let start: u64 = start.parse().ok()?;
            let end: u64 = end.parse().ok()?;
            count.checked_add(end.checked_sub(start)?.checked_add(1)?)
        })
        .filter(|count| *count > 0)
}

#[derive(Debug)]
pub(super) struct CgroupCpuSampler {
    files: Option<CpuFiles>,
    previous: Option<(u64, Instant, f64)>,
}

impl CgroupCpuSampler {
    pub(super) fn new() -> Self {
        let files = fs::read_to_string("/proc/self/cgroup")
            .ok()
            .zip(fs::read_to_string("/proc/self/mountinfo").ok())
            .and_then(|(membership, mounts)| CpuFiles::discover(&membership, &mounts));
        Self {
            files,
            previous: None,
        }
    }

    pub(super) fn sample(&mut self) -> Option<(f64, CpuSource)> {
        self.sample_at(Instant::now())
    }

    fn sample_at(&mut self, now: Instant) -> Option<(f64, CpuSource)> {
        let Some((usage_ns, capacity_cores)) = self.files.as_ref().and_then(CpuFiles::read) else {
            self.previous = None;
            return None;
        };
        let previous = self.previous.replace((usage_ns, now, capacity_cores));
        // A counter reset or a new quota makes the old baseline incomparable.
        let fraction = match previous {
            Some((previous_usage_ns, sampled_at, previous_capacity_cores))
                if usage_ns >= previous_usage_ns
                    && previous_capacity_cores == capacity_cores
                    && now > sampled_at =>
            {
                let used_cpu_ns = (usage_ns - previous_usage_ns) as f64;
                let elapsed_ns = now.duration_since(sampled_at).as_nanos() as f64;
                used_cpu_ns / elapsed_ns / capacity_cores
            }
            _ => 0.0,
        };
        Some((fraction, self.files.as_ref()?.source()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::Duration,
    };

    static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(format!(
                "temporal-cpu-{}-{}",
                std::process::id(),
                NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&directory).unwrap();
            Self(directory)
        }

        fn write(&self, path: &str, value: &str) {
            let path = self.0.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, value).unwrap();
        }

        fn mount(&self, root: &str, path: &str, kind: &str, controllers: &str) -> String {
            format!(
                "1 0 0:1 {root} {} rw - {kind} none rw,{controllers}\n",
                self.0.join(path).display()
            )
        }

        fn v1(&self, separate: bool) -> CgroupCpuSampler {
            self.write("cpu/cpu.cfs_quota_us", "1000000");
            self.write("cpu/cpu.cfs_period_us", "100000");
            self.write(
                if separate {
                    "acct/cpuacct.usage"
                } else {
                    "cpu/cpuacct.usage"
                },
                "0",
            );
            let mounts = if separate {
                self.mount("/host", "cpu", "cgroup", "cpu")
                    + &self.mount("/host", "acct", "cgroup", "cpuacct")
            } else {
                self.mount("/host", "cpu", "cgroup", "cpu,cpuacct")
            };
            let membership = if separate {
                "1:cpu:/\n2:cpuacct:/\n"
            } else {
                "1:cpu,cpuacct:/\n"
            };
            CgroupCpuSampler {
                files: CpuFiles::discover(membership, &mounts),
                previous: None,
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn v1_cpu_quota_counts_whole_cgroup_with_combined_or_separate_mounts() {
        for separate in [false, true] {
            let fixture = Fixture::new();
            let mut sampler = fixture.v1(separate);
            let now = Instant::now();
            assert_eq!(sampler.sample_at(now), Some((0.0, CpuSource::CgroupV1)));
            fixture.write(
                if separate {
                    "acct/cpuacct.usage"
                } else {
                    "cpu/cpuacct.usage"
                },
                "10000000000",
            );
            assert_eq!(
                sampler.sample_at(now + Duration::from_secs(1)),
                Some((1.0, CpuSource::CgroupV1))
            );
        }
    }

    #[test]
    fn v1_uses_nested_membership_and_parent_quota() {
        let fixture = Fixture::new();
        fixture.write("cpu/child/cpuacct.usage", "0");
        fixture.write("cpu/child/cpu.cfs_quota_us", "-1");
        fixture.write("cpu/child/cpu.cfs_period_us", "100000");
        fixture.write("cpu/cpu.cfs_quota_us", "200000");
        fixture.write("cpu/cpu.cfs_period_us", "100000");
        let files = CpuFiles::discover(
            "1:cpu,cpuacct:/host/child\n",
            &fixture.mount("/host", "cpu", "cgroup", "cpu,cpuacct"),
        )
        .unwrap();
        assert_eq!(files.read(), Some((0, 2.0)));
    }

    #[test]
    fn v2_and_hybrid_mounts_select_the_active_cpu_controller() {
        let fixture = Fixture::new();
        let mut sampler = fixture.v1(false);
        let mounts = fixture.mount("/", "unified", "cgroup2", "")
            + &fixture.mount("/host", "cpu", "cgroup", "cpu,cpuacct");
        let membership = "0::/\n1:cpu,cpuacct:/\n";
        assert!(matches!(
            CpuFiles::discover(membership, &mounts),
            Some(CpuFiles::V1 { .. })
        ));
        fixture.write("unified/cpu.stat", "usage_usec 123456\n");
        fixture.write("unified/cpu.max", "50000 100000");
        sampler.files = CpuFiles::discover(membership, &mounts);
        assert_eq!(
            sampler.files.as_ref().unwrap().read(),
            Some((123456000, 0.5))
        );
        let now = Instant::now();
        sampler.sample_at(now);
        fixture.write("unified/cpu.stat", "usage_usec 623456\n");
        assert_eq!(
            sampler.sample_at(now + Duration::from_secs(1)),
            Some((1.0, CpuSource::CgroupV2))
        );
    }

    #[test]
    fn v2_parent_quota_survives_a_root_without_cpu_max() {
        let fixture = Fixture::new();
        fixture.write("cpu/parent/child/cpu.stat", "usage_usec 100");
        fixture.write("cpu/parent/child/cpu.max", "max 100000");
        fixture.write("cpu/parent/cpu.max", "150000 100000");
        let files = CpuFiles::discover(
            "0::/parent/child\n",
            &fixture.mount("/", "cpu", "cgroup2", ""),
        )
        .unwrap();
        assert_eq!(files.read(), Some((100000, 1.5)));
        fixture.write("cpu/parent/cpu.max", "invalid");
        assert!(files.read().is_none());
    }

    #[test]
    fn escaped_mount_and_nonexistent_bind_mount_are_resolved() {
        let fixture = Fixture::new();
        fixture.write("cpu space/cpuacct.usage", "42");
        fixture.write("cpu space/cpu.cfs_quota_us", "100000");
        fixture.write("cpu space/cpu.cfs_period_us", "100000");
        let mounts = fixture.mount("/", "missing", "cgroup", "cpu,cpuacct")
            + &fixture
                .mount("/", "cpu space", "cgroup", "cpu,cpuacct")
                .replace("cpu space", "cpu\\040space");
        let files = CpuFiles::discover("1:cpu,cpuacct:/\n", &mounts).unwrap();
        assert_eq!(files.read(), Some((42, 1.0)));
    }

    #[test]
    fn cpuset_can_reduce_a_finite_cpu_quota() {
        let fixture = Fixture::new();
        let mut sampler = fixture.v1(false);
        fixture.write("set/cpuset.cpus", "0-1,4");
        let mounts = fixture.mount("/host", "cpu", "cgroup", "cpu,cpuacct")
            + &fixture.mount("/host", "set", "cgroup", "cpuset");
        sampler.files = CpuFiles::discover("1:cpu,cpuacct:/\n2:cpuset:/\n", &mounts);
        assert_eq!(sampler.files.as_ref().unwrap().read(), Some((0, 3.0)));
    }

    #[test]
    fn invalid_and_unlimited_quotas_do_not_produce_container_percentages() {
        let fixture = Fixture::new();
        let mut sampler = fixture.v1(false);
        for quota in ["-1", "0", "bad"] {
            fixture.write("cpu/cpu.cfs_quota_us", quota);
            assert!(sampler.sample_at(Instant::now()).is_none());
        }
        fixture.write("cpu/cpu.cfs_quota_us", "100000");
        fixture.write("cpu/cpu.cfs_period_us", "0");
        assert!(sampler.sample_at(Instant::now()).is_none());
    }

    #[test]
    fn v2_unlimited_or_invalid_limits_reset_the_sample() {
        let fixture = Fixture::new();
        fixture.write("cpu/cpu.stat", "usage_usec 1000000\n");
        fixture.write("cpu/cpu.max", "100000 100000");
        let mut sampler = CgroupCpuSampler {
            files: CpuFiles::discover("0::/\n", &fixture.mount("/", "cpu", "cgroup2", "")),
            previous: None,
        };
        let mut now = Instant::now();
        sampler.sample_at(now);

        for limit in ["max 100000", "0 100000", "100000 0", "invalid"] {
            fixture.write("cpu/cpu.max", limit);
            now += Duration::from_secs(1);
            assert_eq!(sampler.sample_at(now), None, "limit: {limit}");

            fixture.write("cpu/cpu.max", "100000 100000");
            now += Duration::from_secs(1);
            assert_eq!(sampler.sample_at(now), Some((0.0, CpuSource::CgroupV2)));
        }

        fixture.write("cpu/cpu.stat", "usage_usec 2000000\n");
        assert_eq!(
            sampler.sample_at(now + Duration::from_secs(1)),
            Some((1.0, CpuSource::CgroupV2))
        );
    }

    #[test]
    fn counter_reset_quota_change_and_read_failure_reset_the_baseline() {
        let fixture = Fixture::new();
        let mut sampler = fixture.v1(false);
        let now = Instant::now();
        sampler.sample_at(now);
        fixture.write("cpu/cpuacct.usage", "10000000000");
        assert_eq!(
            sampler.sample_at(now + Duration::from_secs(1)).unwrap().0,
            1.0
        );
        fixture.write("cpu/cpuacct.usage", "0");
        assert_eq!(
            sampler.sample_at(now + Duration::from_secs(2)).unwrap().0,
            0.0
        );
        fixture.write("cpu/cpu.cfs_quota_us", "2000000");
        fixture.write("cpu/cpuacct.usage", "10000000000");
        assert_eq!(
            sampler.sample_at(now + Duration::from_secs(3)).unwrap().0,
            0.0
        );
        fs::remove_file(fixture.0.join("cpu/cpuacct.usage")).unwrap();
        assert!(sampler.sample_at(now + Duration::from_secs(4)).is_none());
        fixture.write("cpu/cpuacct.usage", "20000000000");
        assert_eq!(
            sampler.sample_at(now + Duration::from_secs(5)).unwrap().0,
            0.0
        );
        fixture.write("cpu/cpuacct.usage", "40000000000");
        assert_eq!(
            sampler.sample_at(now + Duration::from_secs(6)).unwrap().0,
            1.0
        );
    }
}
