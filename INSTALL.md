# Install python requirements
```bash
pip install -r requirements/res.txt --no-deps
```


# Lock python requirements
```bash
pip-compile -v --output-file requirements/res.txt requirements/res.in
```
