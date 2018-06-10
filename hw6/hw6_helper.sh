#!/bin/bash

OPTION=$1
USERNAME=$2
PASSWORD=$3
SEND_PATH=$4
INPUT_SIZE=$5
FILTER_SIZE=$6
ALPHA=$7
BETA=$8
JOB_ID_DOWNLOAD=$5
JOB_ID=$4
HOSTNAME=chundoong0.snu.ac.kr

usage(){
	echo -e "Usage : "
	echo -e "send code and compile remotley and remove code and lastly enqueue the job at once"
	echo -e "\t $0 remote_run [username] [password] [send_path] [input_size] [filter_size] [alpha] [beta]"
	echo -e "\t example) $0 remote_run hyu00 Y0URP@SSW0RD test 15 2 0.5 2"
	echo -e  
	echo -e "download result file from login node"
	echo -e "\t $0 download_result [username] [password] [sent_path] [job_id]"
	echo -e "\t example) $0 download_result hyu00 y0urp@ssw0rd 818670"
	echo -e "\t sent_path should be same with remote_run's one"
	echo -e  
	echo -e "check job status"
	echo -e "\t $0 stat [username] [password] [job_id]"
	echo -e "\t example) $0 stat hyu00 y0urp@ssw0rd 818670"
	echo -e  
	echo -e "kill job"
	echo -e "\t $0 kill [username] [password] [job_id]"
	echo -e "\t example) $0 kill hyu00 y0urp@ssw0rd 818670"
}

send(){
	sshpass -p$PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOSTNAME "mkdir -p $SEND_PATH"
	sshpass -p$PASSWORD scp -o StrictHostKeyChecking=no ./* $USERNAME@$HOSTNAME:/home/$USERNAME/$SEND_PATH
}

remove(){
	sshpass -p$PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOSTNAME "rm $SEND_PATH/*.cu"
}

remote_compile(){
	sshpass -p$PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOSTNAME "cd $SEND_PATH;make clean;make all"
}

send_compile_remove(){
	send
	remote_compile
	remove
}

stat(){
	sshpass -p$PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOSTNAME "thorq --stat $JOB_ID"
}

kill(){
	sshpass -p$PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOSTNAME "thorq --kill $JOB_ID"
}

remote_run(){
	send_compile_remove
	sshpass -p$PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOSTNAME "cd $SEND_PATH;thorq --add --mode single --device gpu/1080 --timeout 300 ./hw6 $INPUT_SIZE $FILTER_SIZE $ALPHA $BETA"
}

download_result(){
	sshpass -p$PASSWORD scp -o StrictHostKeyChecking=no $USERNAME@$HOSTNAME:/home/$USERNAME/$SEND_PATH/task_$JOB_ID_DOWNLOAD.std* ./
	
	echo ========================================== stdout ==========================================
	cat task_$JOB_ID_DOWNLOAD.stdout
	echo -e "\n\n\n\n"
	
	echo ========================================== stderr ==========================================
	cat task_$JOB_ID_DOWNLOAD.stderr
}



case $OPTION in
	"send") send
	;;
	"remove") remove
	;;
	"remote_compile") remote_compile
	;;
	"send_compile_remove") send_compile_remove
	;;
	"remote_run") remote_run
	;;
	"download_result") download_result
	;;
	"stat") stat
	;;
	"kill") kill
	;;
	*) usage
esac
