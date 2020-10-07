#!/bin/bash
for j in {12..25}
do
	a=$(expr $j \* 10)
	b=$(expr $a + 7)
	c=$(expr $b + 1)
	d=$(expr $c + 1)
	for ((i=$a; i<=$b; i++))
	do
		mv ~/Downloads/Olivetti-PNG-master/images/image-$i.png person$j
	done
	for ((i=$c; i<=$d; i++))
	do
		mv ~/Downloads/Olivetti-PNG-master/images/image-$i.png person$j-test
	done
done
