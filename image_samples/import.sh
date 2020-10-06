#!/bin/bash
j=11
mkdir person$j
mkdir person$j-test
	for i in {110..117}
	do
		mv ~/Downloads/Olivetti-PNG-master/images/image-$i.png person$j
	done
	for i in {118..119}
        do
                mv ~/Downloads/Olivetti-PNG-master/images/image-$i.png person$j-test
        done
