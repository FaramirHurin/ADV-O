#!/bin/bash 

nvidia-docker run \
--rm \
--tty \
--interactive \
--detach \
--name advo_docker \
--memory=400G \
--memory-swap=450G \
--shm-size=50G  \
--volume "/media/hdd3/worldline_home:/app/worldline_home:Z" \
--volume "/media/hdd3/dlunghi_docker_volume/CleanPipeline:/app/CleanPipeline:Z" \
--volume "/home/gpaldino/git/ADV-O:/app/ADV-O:Z" \
--publish 9999:9999 \
advo-image
