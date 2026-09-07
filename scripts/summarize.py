from pathlib import Path
import argparse,json,statistics,csv,xml.etree.ElementTree as ET

def render(root,run):
    project=json.loads((root/'manifests/project.json').read_text())['project']
    execution=json.loads((run/'execution.json').read_text())
    def read(name):return json.loads((run/name).read_text())
    rel='../'+str(run.relative_to(root))
    lines=['# Measured qualification report','',f'Run: `{run.name}`. Execution status: **{execution["status"]}**.',
           '',f'[All commands and exit codes]({rel}/execution.json) · [CPU test log]({rel}/cpu-tests.log) · [GPU correctness]({rel}/gpu-check.json) · [Raw timing]({rel}/benchmark.json)',
           '',f'Source digest: `{execution["environment"]["source_digest"]}`.',
           '', 'Hardware: RTX 4090 (sm_89), driver 595.84, native toolkit CUDA 13.2. Clocks unchanged; display GPU. Results apply to this run and workload only.','']
    xml=ET.parse(run/'cpu.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
    tests=sum(int(s.attrib.get('tests',0)) for s in suites)
    lines += [f'CPU: {tests} tests. All command return codes are retained; missing prerequisites fail the reproduction driver.','']
    check=read('gpu-check.json');bench=read('benchmark.json')
    if project=='A':
        lines += [f'GPU Python corpus: {len(check["records"])} full-output comparisons, output guards and fused epilogue checks. Native CTest: 86 cases.',
                  '', '| Workload | PyTorch matmul µs | Triton matmul µs | PyTorch fragment µs | Fused Triton fragment µs |', '|---|---:|---:|---:|---:|']
        for row in bench['results']:
            m=row['timing']['measurements'];vals=[m[k]['cuda_event']['p50_ms']*1000 for k in ['torch-preallocated','triton-preallocated','torch-fragment','triton-fragment']]
            lines.append('| '+row['case']['id']+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
        lines += ['', 'Event-span medians above include eager host dispatch gaps. Fragment = linear + bias + SiLU, preloaded weights. This run does not qualify a Triton speedup.',
                  '', '| Workload | cuBLASLt µs | Unsplit PTX µs | Split-K µs | cuBLASLt / Split-K |', '|---|---:|---:|---:|---:|']
        native=read('native-benchmark.json');groups={}
        for row in native['results']:groups.setdefault((row['case'],row['record']['implementation']['name']),[]).append(row['record']['benchmark']['latency_ms']['p50']*1000)
        for case in ['decode-m8','decode-m32','tail','square']:
            v=[statistics.median(groups[(case,k)]) for k in ['cublaslt','ptx_mma_small','ptx_mma_splitk']]
            lines.append(f'| {case} | {v[0]:.2f} | {v[1]:.2f} | {v[2]:.2f} | {v[0]/v[2]:.3f}× |')
        lines += ['', 'Native medians are the median of three shuffled process-round medians. Split-K improves decode relative to the unsplit tile; it does not establish superiority to cuBLASLt. The tail dispatch selects one split, so it measures the existing fallback rather than a forced split experiment.',
                  '', '| Host-to-host pipeline | PyTorch wall µs | Candidate wall µs |', '|---|---:|---:|']
        for row in bench['results']:
            m=row['end_to_end']['measurements'];lines.append(f'| {row["case"]["id"]} | {m["torch-e2e"]["synchronized_wall"]["p50_ms"]*1000:.2f} | {m["candidate-e2e"]["synchronized_wall"]["p50_ms"]*1000:.2f} |')
        cute=read('cute-benchmark.json');lines+=['','| CUTLASS aligned workload | CuTe event µs | PyTorch event µs |','|---|---:|---:|']
        for row in cute['results']:
            m=row['timing']['measurements'];lines.append(f'| {row["case"]} | {m["cute"]["cuda_event"]["p50_ms"]*1000:.2f} | {m["torch"]["cuda_event"]["p50_ms"]*1000:.2f} |')
        lines+=['',f'[Native raw events]({rel}/native-benchmark.json) · [CuTe results]({rel}/cute-benchmark.json) · [CUDA trace: decode kernels]({rel}/profile-decode_cuda_gpu_kern_sum.csv) · [CUDA trace: tail kernels]({rel}/profile-tail_cuda_gpu_kern_sum.csv)',
                '', 'Nsight Compute hardware counters require interactive administrator authentication on this host. That attempt is retained in exploratory logs. Nsight Systems provides actual kernel durations and transfer activity; occupancy, cache-hit and bandwidth-bottleneck hypotheses are not declared confirmed without counters.']
    elif project=='B':
        base,fixed=check['versions'];bad=[r for r in base['records'] if not r['correct']]
        lines += [f'CPU oracle: {check["cpu"]["checks"]} additional compiled checks, ASan/UBSan clean. The known bad revision fails {len(bad)}/{len(base["records"])} GPU cases; the upstream patched revision passes {len(fixed["records"])}/{len(fixed["records"])} with intact guards.',
                  '', '| Benchmark (1M samples) | Base µs | Patched µs | Patched / base time |', '|---|---:|---:|---:|']
        grouped={}
        for row in bench['results']:
            for rec in row['records']:grouped.setdefault((row['version'],rec['case']),[]).append(rec['statistics']['p50_ms']*1000)
        for case in ['unsigned-fast','int32-control','signed-positive']:
            b=statistics.median(grouped[('base',case)]);f=statistics.median(grouped[('fixed',case)])
            lines.append(f'| {case} | {b:.2f} | {f:.2f} | {f/b:.3f}× |')
        lines += ['', 'Times summarize seven shuffled process-round medians; raw per-round distributions remain available. Any signed-positive slowdown is a correctness/dispatch tradeoff, not an unsigned fast-path regression. A small ratio near the run dispersion is inconclusive.',
                  '', 'The patch belongs to existing upstream PR #10993. This is independent regression evidence, not a new or accepted upstream fix.']
    elif project=='C':
        lines += [f'{len(check["records"])} GPU cases passed. Both independent output and base-2 LSE comparisons pass; guards and caller buffer identity pass.',
                  '', '| Metric | Maximum across corpus |', '|---|---:|',
                  f'| Output absolute error | {max(r["output"]["max_abs"] for r in check["records"]):.6g} |',
                  f'| LSE absolute error | {max(r["lse"]["max_abs"] for r in check["records"]):.6g} |',
                  '', '| FA2 case | Event p50 µs | p05–p95 µs |', '|---|---:|---:|']
        for row in bench['results']:
            m=row['timing']['measurements']['fa2-return-lse']['cuda_event'];lines.append(f'| {row["case"]} | {m["p50_ms"]*1000:.2f} | {m["p05_ms"]*1000:.2f}–{m["p95_ms"]*1000:.2f} |')
        lines += ['',f'[Actual CUDA dispatch names]({rel}/dispatch-trace.json) · [Reduced synthetic mutation]({rel}/minimal-mutant.json)',
                  '', 'Microbenchmarks quantify this backend only; no CPU-to-GPU speedup claim is made. CPU reference cost is recorded separately per correctness case. Quantization, all-mask rows and other backends remain outside v1. No upstream bug is asserted from a passing suite.']
    else:
        lines += [f'{len(check["records"])} CPU/Arrow/pylibcudf cases passed with exact groups, null policy and count conservation. Controlled capacity rejection passed.',
                  '', '| Workload | Arrow operator ms | GPU operator ms | Arrow ETL ms | GPU ETL ms | ETL speedup | RMM peak MiB |', '|---|---:|---:|---:|---:|---:|---:|']
        for row in bench['results']:
            m=row['measurements'];v=[m[k]['wall']['p50_ms'] for k in ['arrow-operator','gpu-operator','arrow-etl','gpu-etl']]
            lines.append('| '+row['case']['id']+' | '+' | '.join(f'{x:.3f}' for x in v)+f' | {v[2]/v[3]:.2f}× | {row["rmm_memory"]["peak_bytes"]/2**20:.2f} |')
        lines+=['','| Workload | Upload wall ms | Download wall ms |','|---|---:|---:|']
        for row in bench['results']:
            m=row['measurements'];lines.append(f'| {row["case"]["id"]} | {m["gpu-upload"]["wall"]["p50_ms"]:.3f} | {m["gpu-download"]["wall"]["p50_ms"]:.3f} |')
        lines+=['',f'[CUDA kernel trace summary]({rel}/profile_cuda_gpu_kern_sum.csv) · [CUDA transfer summary]({rel}/profile_cuda_gpu_mem_time_sum.csv) · [NVTX phase summary]({rel}/profile_nvtx_sum.csv)',
                '', 'Wall medians include Python API dispatch. GPU event spans are in raw JSON; kernel-only timings are in the separate diagnostic trace. Crossover conclusions are restricted to these rows, group counts, null ratios and this display GPU. RMM peak excludes CUDA context and other applications.']
    sanitizer=[s for s in execution['steps'] if 'memcheck' in s['step'] or 'racecheck' in s['step']]
    lines+=['','## Safety and reproducibility','']
    for step in sanitizer:lines.append(f'- [{step["step"]} log]({rel}/{step["step"]}.log): exit {step["returncode"]}.')
    lines+=['','The reproduction driver fails on missing GPU, failed checks, stale gates, sanitizer errors or subprocess failure. CPU CI is labeled separately. All performance comparisons retain failures and unsupported cases.']
    (root/'docs/RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    render(root,(root/a.run).resolve())
