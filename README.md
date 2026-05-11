# iDMRG

An infinite DMRG library built on ITensors.jl and ITensorMPS.jl.

## Installation

The package is not yet registered. To install, use the Julia Pkg REPL:

```
pkg> add https://github.com/liuzhaochen/iDMRG
```

## Limitations

- The MPO is not directly constructed. The workaround: build a finite-size MPO via ITensorMPS and assume the central part is translation invariant.
- Environment fixed points are solved via power iteration with DIIS (Anderson) acceleration.

## Dependencies

ITensors.jl, ITensorMPS.jl, KrylovKit.jl.

## Author

Zhaochen Liu — zhaochen.liu@uni-wuerzburg.de

## Citation

If you use this package in your research, please cite:

```bibtex
@software{liu2025idmrg,
  author = {Liu, Zhaochen},
  title = {iDMRG.jl: A Julia Package for Infinite Density Matrix Renormalization Group},
  year = {2025},
  url = {https://github.com/liuzhaochen/iDMRG.jl}
}
```

## License

MIT License
