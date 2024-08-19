test:
	python -m pytest tests/test_source_code.py

integration_test:
	cd ./tests && sh ./run_integration_test.sh && cd ..

quality_checks:
	isort .
	black .
	pylint --recursive=y . ; echo $$?

setup:
	# Pipenv
	pip install --upgrade pip
	pip install pipenv==2024.0.1
	pipenv install --dev
	pipenv install backports.tarfile #For compatibility reasons

	# Pre-commit hooks
	pre-commit install

build_docker_container: quality_checks test integration_test
	docker build -t bank-churn-prediction-service:v1 -f deployment/Dockerfile .

run_docker_container: build_docker_container
	docker run -it --rm -p 9696:9696 bank-churn-prediction-service:v1


