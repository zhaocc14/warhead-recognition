#### The function of each `.m` file

* `genMDmap.m` generates micro-Dopple map by `RD_Data/OMP_Data/`
* `genRDomp_d.m` and `genRDomp_w.m` generate `RD_Data/OMP_Data/` by OMP with `ECHO_Data/`, where 'd' denotes 'Debris' and 'w' denotes 'Warhead' 
  * `config.m` configs some constants
  * `get_OMPmap_from_echo.m` is the exact functional function
  * `omp.m` is the exact function which realize OMP algrithom
