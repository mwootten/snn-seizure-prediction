important: multi-spiking-weights-sans2.json

graphs: spiking-graph.png traditional-graph.png

spiking-graph.png: spiking-true spiking-predicted
	python3 ../postprocessing/post-process.py --predicted spiking-predicted --true spiking-true --output spiking-graph.png

traditional-training.json traditional-testing.json: marker.txt
	python3 ../preprocessing/neural/network.py --iterations 1000 --training traditional-training.json --test traditional-testing.json ../preprocessing/fourier-transform/out/*/*/*.freq32

traditional-sans2-training.json: traditional-training.json
	jq '.[1]' $< > $@

traditional-sans1-training.json: traditional-training.json
	jq '.[2]' $< > $@

traditional-sans0-training.json: traditional-training.json
	jq '.[3]' $< > $@

traditional-sans2-testing.json: traditional-testing.json
	jq '.[1]' $< > $@

traditional-sans1-testing.json: traditional-testing.json
	jq '.[2]' $< > $@

traditional-sans0-testing.json: traditional-testing.json
	jq '.[3]' $< > $@

spiking-training.json: traditional-training.json
	python3 ../preprocessing/bridge/real-to-spikes.py traditional-training.json > spiking-training.json

spiking-sans2-training.json: traditional-sans2-training.json
	python3 ../preprocessing/bridge/real-to-spikes.py $< > $@

spiking-sans2-testing.json: traditional-sans2-testing.json
	python3 ../preprocessing/bridge/real-to-spikes.py $< > $@

spiking-testing.json: traditional-testing.json
	python3 ../preprocessing/bridge/real-to-spikes.py traditional-testing.json > spiking-testing.json

schedule-training.pickle: spiking-training.json
	python3 ../preprocessing/bridge/schedule.py spiking-training.json schedule-training.pickle

schedule-testing.pickle: spiking-testing.json
	python3 ../preprocessing/bridge/schedule-for-testing.py spiking-testing.json schedule-testing.pickle

schedule-sans2-training.pickle: spiking-sans2-training.json
	python3 ../preprocessing/bridge/schedule.py $< $@

schedule-sans2-testing.pickle: spiking-sans2-testing.json
	python3 ../preprocessing/bridge/schedule-for-testing.py $< $@

multi-spiking-weights.json: schedule-training.pickle
	pypy3 ../multi-spiking/multi-spiking-standard.py schedule-training.pickle 61 4 > multi-spiking-weights.json

multi-spiking-weights-sans2.json: schedule-sans2-training.pickle
	pypy3 ../multi-spiking/multi-spiking-standard.py $< 61 4 > $@

multi-spiking-weights-sans1.json: schedule-sans1-training.pickle
	pypy3 ../multi-spiking/multi-spiking-standard.py $< 21 3 > $@

spiking-test-results.json: multi-spiking-weights.json schedule-testing.pickle
	pypy3 ../multi-spiking/runNetwork.py --weights multi-spiking-weights.json schedule-testing.pickle > spiking-test-results.json

spiking-test-results-sans2.json: multi-spiking-weights-sans2.json schedule-sans2-testing.pickle
	pypy3 ../multi-spiking/runNetwork.py --weights multi-spiking-weights-sans2.json schedule-sans2-testing.pickle > $@

distinct-results.json: spiking-test-results.json
	cat spiking-test-results.json | sort | awk '!(NR%500)' > distinct-results.json

spiking-true: distinct-results.json
	cat $< | jq -sr 'map(.[1] | (. / 5) - 3 | tostring) | join("\n")' > $@

spiking-sans2-true: spiking-test-results-sans2.json
	cat $< | jq -sr 'map(.[1] | (. / 5) - 3 | tostring) | join("\n")' > $@

spiking-predicted: distinct-results.json
	cat $< | jq -sr 'map(.[0] | tostring) | join("\n")' > $@

spiking-sans2-predicted: spiking-test-results-sans2.json
	cat $< | jq -sr 'map(.[0] | tostring) | join("\n")' > $@

traditional-predictions: trad-marker.txt
	python3 ../preprocessing/neural/network.py --iterations 1000 --without 0 --training /dev/null --test traditional-predictions ../preprocessing/fourier-transform/out/*/*/*.freq32

traditional-true: trad-marker.txt
	ls ../preprocessing/fourier-transform/out/test/positive/*.freq32 | wc -l | jq -r '(([range(.)] | map(0)) + ([range(.)] | map(1))) | map(tostring) | join("\n")' > traditional-true

traditional-graph.png: traditional-predictions traditional-true
	python3 ../postprocessing/post-process.py --predicted traditional-predictions --true traditional-true --output traditional-graph.png

clean:
	rm *.pickle *.json *.png

traditional-predictions-new: traditional-testing.json
	jq -r '.[3] | map(.[0]) | map(tostring) | join("\n")' traditional-testing.json > traditional-predictions-new
