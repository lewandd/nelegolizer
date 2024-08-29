init:
	pip install -r requirements.txt --no-cache-dir

install:
	pip install -e .

test:
	python3 -m unittest discover -s tests/unit/
	python3 -m unittest discover -s tests/integration/