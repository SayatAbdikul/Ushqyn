# Decision-round evidence archive

This pack preserves the completed 1 October 2026 factor, deadline-policy and resident-controller feasibility experiments. The native and trace evidence passes independent audit; the resident FPGA candidate fails placement, while its matched baseline routes. Failed diagnostics are retained with their successful replacements clearly identified.

evidence.tar.gz contains source snapshots, frozen selections, native input/output/program fixtures, raw schedule and native reports, compiler/control dependencies, matched FPGA recipes/resource/timing reports and logs. manifest.json records each member's size and SHA-256. verification.json records a successful streamed verification of every archive member and the archive digest.

The archive includes original production-source snapshots and copies of external control/compiler/FPGA sources, avoiding dependence on an ephemeral worktree for evidence inspection. Factor source snapshots regenerate both selected command/payload binaries exactly. The archive excludes original ONNX/dataset maps, proprietary tools, generated native objects/executables, FPGA netlists and bitstreams. Fresh experiments require the documented toolchain and original benchmark inputs. Archived reports retain their original absolute provenance paths; commands are not automatically relocated for a different workstation. Evidence integrity can be verified without executing the experiments.

From the original project root, recreate the pack with:

```
.venv/bin/python3 tools/phase6/archive_decision_round.py
```

See docs/research/PHASE_6_IMPLEMENTATION_DECISION_ROUND_2026_10_01.md inside the archive for conclusions, comparison tables and limitations. No Q1 novelty or worst-case schedulability is established by these experiments.
