CUR_DIR=$(cd $(dirname $0) && pwd)

LOG_DIR=$CUR_DIR/log
PID_DIR=$CUR_DIR/pid

CHATBOT_PIDS=`ps -ef | grep streamlit | grep chatbot | awk '{print $2}'`
echo $CHATBOT_PIDS
if [ -n "$CHATBOT_PIDS" ]; then 
    echo killing running chatbot app...
    for pid in $CHATBOT_PIDS; do
        echo killing pid $pid
        kill $pid
    done
fi

LOGPARSER_PIDS=`ps -ef | grep log_parser | grep -i python | awk '{print $2}'`
echo $LOGPARSER_PIDS
if [ -n "$LOGPARSER_PIDS" ]; then 
    echo killing running log parser...
    for pid in $LOGPARSER_PIDS; do
        echo killing log parser $pid
        kill $pid
    done
fi


stop_all(){
    echo stoppoing chatbot...
    kill $CHATBOT_PID $LOGPARSER_PID
    wait $CHATBOT_PID $LOGPARSER_PID
    echo 'processes stopped'
}
trap 'stop_all' SIGINT


echo starting chatbot...
streamlit run $CUR_DIR/chatbot_logging.py &
CHATBOT_PID=`echo $!`
echo chatbot is running on $CHATBOT_PID

sleep 10

echo starting log parser...
python log_parser.py &
LOGPARSER_PID=`echo $!`
echo log parser is running on $LOGPARSER_PID

wait $CHATBOT_PID $LOGPARSER_PID

