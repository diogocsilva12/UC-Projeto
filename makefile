.PHONY: install run clean

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run:
	python audio_benchmark.py --audio_dir=audio_data --max_files=10000 --repeat=1

clean:
	rm -f resultados_benchmark.csv benchmark_audio.png