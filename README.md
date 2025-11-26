To get started, create a virtual environment:

```bash
python -m venv .venv
```

This will create a virtual environment, with name .venv. For the time being, there is no `requirements.txt` file, so you'll have to import stuff manually. Moreover, you should keep in mind that the `pyIGRF` library has a coefficient file that does not autotmatically appear, hence you will get an error. Just Google the file, and add it at its specified location. 

If you want to run the filter, you will need some synthetic data. Hence, start by running the following command:
```bash
python -m data.generator
```

This will generate data according to an enviromental model and orbital propagator, and store it in a SQLite database.

Now proceed with the following command:

```bash
python -m sim.run
```

Which will store data in the same database. You can plot this by running 
```bash
python -m plotting.attitude_plotter
```

That is all for now. 