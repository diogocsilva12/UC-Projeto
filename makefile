.PHONY: install run clean

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run:
	python audio_benchmark.py

clean:
	rm -f resultados_benchmark.csv benchmark_audio.png