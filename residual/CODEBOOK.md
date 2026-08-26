# Frozen sphere codebook

The seams patch carries only the *header* of `vllm/poc/sphere_codebook.pt`
(`git diff` without `--binary`), so `patch` never creates the file, and
`get_sphere_codebook()` rebuilds the codebook with Adam — warning only
(`NOT consensus-safe`). Apply the patch, then:

```bash
SP=$(python -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')
cp residual/sphere_codebook.pt "$SP/vllm/poc/sphere_codebook.pt"
sha256sum "$SP/vllm/poc/sphere_codebook.pt"
# 73d57b52b284c0efb0aaabe47707fd5d281d70abbd58fd04a6c5766a74a3c4b9
```

Same bytes the plugin package ships — the production path is unaffected.
