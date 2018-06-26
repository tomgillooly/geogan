#!/bin/bash

# for model in `cat models_to_test`
# do
# 	echo $model

# 	if [ ! -e checkpoints/$model ]; then
# 		mkdir checkpoints/$model
# 	fi

# 	for file in `ssh oggy "egrep -l $model" geo_gan.git/slurm*`
# 	do
# 		scp oggy:$file checkpoints/$model/
# 	done
# 	scp oggy:geo_gan.git/checkpoints/$model/latest* checkpoints/$model/

# 	scp oggy:geo_gan.git/checkpoints/$model/opt.txt checkpoints/$model/

# 	scp oggy:geo_gan.git/checkpoints/$model/loss_log.txt checkpoints/$model/

# 	if [ ! -e checkpoints/$model/latest_net_G.pth ]; then
# 		exit 1
# 	fi
# done
if [ $HOSTNAME == 'liris501-r073' ]; then
	for model in `cat models_to_test`
	do
		# echo $model

		egrep -l $model slurm* | xargs -n1 -I{} cp {} checkpoints/$model/

		cp checkpoints/$model/opt.txt{,.train}
	done
fi

for model in `cat models_to_test`
do
	echo $model
	bash test_circle.sh latest $model
	python assorted_test/plot_loss.py $model
done

# for model in `cat models_to_test`
# do

# 	scp oggy:geo_gan.git/checkpoints/$model/opt.txt checkpoints/$model/

# done