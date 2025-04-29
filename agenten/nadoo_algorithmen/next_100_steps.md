# Next 100 Small Steps Roadmap

1. Fix import error in `run_block_chunked_synthetic.py`.
2. Add `nadoo_algorithmen` to `PYTHONPATH` in experiment scripts.
3. Rerun Block-Chunked synthetic MLP experiments.
4. Parse updated results and update paper section 5.1.
5. Automate generation of summary tables from CSV outputs.
6. Add unit tests for `split_weights.py`.
7. Fix file paths in `run_activity_routing_synthetic.py`.
8. Rerun synthetic Activity Routing experiments.
9. Verify skip_ratio computation and quartile metrics.
10. Update Activity Routing paper section 5.1 with new metrics.
11. Add memory profiling to synthetic activity routing scripts.
12. Integrate energy measurement using `pyRAPL` where available.
13. Rerun classification Activity Routing on MNIST with energy logging.
14. Update paper section 5.2 with complete metrics.
15. Adjust threshold τ values for activity routing: 0.05, 0.1, 0.2.
16. Perform ablation on threshold effect on skip_ratio and latency.
17. Update results discussion in Activity Routing paper.
18. Create hyperparameter sweep script for block-chunk quantization.
19. Run synthetic MLP for k in {1,2,3,4} and quant thresholds τ={0.2,0.5}.
20. Record memory vs. speed trade-offs for each configuration.
21. Update Block-Chunked paper with k/τ ablation table.
22. Optimize chunk-loading code to reduce overhead by 5%.
23. Benchmark optimized code for latency and memory.
24. Add docstrings and type hints to core modules.
25. Write unit tests for `quantize` function in block_chunk module.
26. Fix deprecation warnings in torchvision calls.
27. Update boundary pruning scripts to use consistent dataset loaders.
28. Rerun boundary pruning synthetic experiments (5 runs).
29. Collect and record pruning results and memory usage.
30. Update Boundary Pruning paper section 5.1 with metrics.
31. Sweep pruning threshold τ_prune in {0.1,0.2,0.3}.
32. Measure parameter retention and latency for each τ_prune.
33. Plot retention vs. accuracy curves for boundary pruning.
34. Update paper discussion with boundary pruning analysis.
35. Integrate dynamic quantization in boundary pruning experiments.
36. Rerun boundary pruning with quant-switch mechanism.
37. Document performance of quant-switch in paper section 5.2.
38. Refine pseudocode clarity in all papers.
39. Add initial unit tests for `streaming_kv_inference`.
40. Implement `run_streaming_inference.py` experiments script.
41. Measure token-level latency and memory for window sizes {128,256,512}.
42. Plot latency vs. window_size curves for streaming inference.
43. Update Streaming Inference paper section 5 with initial results.
44. Sweep simulated communication overhead T_comm values.
45. Compare analytical model predictions to empirical throughput.
46. Validate analytical model vs. measured results.
47. Update theoretical section with empirical data.
48. Write unit tests for scheduling and caching modules.
49. Mock model chunks to test scheduler under load.
50. Develop prototype dispatch server for volunteer computing.
51. Implement worker script for volunteer participant nodes.
52. Simulate volunteer network with 3 local worker processes.
53. Measure scheduling latency and completion rates.
54. Add fault-tolerance test for straggler mitigation (replication factor 2).
55. Log straggler recovery overhead and update paper section 3.10.
56. Implement AES-256 chunk encryption and measure overhead.
57. Integrate homomorphic aggregation proof-of-concept and benchmark.
58. Benchmark secure aggregation overhead vs. baseline.
59. Update Security & Privacy section with overhead metrics.
60. Prototype incentive model: simulate energy cost vs. reward.
61. Evaluate incentive model with synthetic reward schedules.
62. Add cost model simulation script (`simulate_incentives.py`).
63. Compare user cost C_user across scenarios and document.
64. Document incentive model results in paper section 3.12.
65. Write integration tests for scheduler and aggregation components.
66. Create Dockerfile for consistent experiment environment.
67. Build and push Docker image to registry.
68. Update README with Docker usage instructions.
69. Set up CI pipeline to run unit tests and linting on push.
70. Add CI workflow to run key experiments on PR.
71. Containerize Jupyter notebooks for reproducibility.
72. Implement experiment configuration via YAML files.
73. Refactor experiment scripts to accept config inputs.
74. Validate experiments with new config loading.
75. Write script to generate LaTeX tables from CSV results.
76. Generate combined results report in Markdown and PDF.
77. Create baseline comparison plots for all methods.
78. Aggregate all plots in the `analysis/` directory.
79. Update main README with links to analysis artifacts.
80. Review and unify citation formats across all papers.
81. Add missing references for streaming and volunteer computing.
82. Check for broken links in all Markdown files.
83. Ensure all code files have license headers.
84. Add contribution guidelines and code of conduct.
85. Draft email to collaborators summarizing current status.
86. Schedule team meeting to review experimental roadmap.
87. Create issue tracker entries for remaining tasks.
88. Update memory database with current experiment metadata.
89. Automate memory creation for new milestones.
90. Run code linting and fix style issues.
91. Run `pytest` with coverage reporting and badges.
92. Add test coverage badge to repository README.
93. Document known limitations and edge cases in papers.
94. Collect feedback from initial reviewers and stakeholders.
95. Incorporate reviewer comments into drafts.
96. Prepare preprint versions for arXiv submission.
97. Format papers to IEEE and ACL style templates.
98. Generate PDF proofs for each paper and review.
99. Perform final sanity check on all results and code.
100. Consolidate final recommendations and submit manuscripts.
