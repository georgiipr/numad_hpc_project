Below are the commands to run the repo (MacOS). All pathes are viable for my system only.

compilation of cpp file:
```
g++ -std=c++17 -Xpreprocessor -fopenmp -lomp -I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib /Users/egor/Downloads/HPC_numad/model_numadpars.cpp -o /Users/egor/Downloads/HPC_numad/model_numadpars
```

launch of cpp file:
```
/Users/egor/Downloads/HPC_numad/model_numadpars
```

launch of .py file (plotter):
```
/usr/local/bin/python3 /Users/egor/Downloads/HPC_numad/numad_hpc_plotter.py
```
