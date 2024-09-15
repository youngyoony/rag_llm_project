import json
import time
from kafka import KafkaProducer, KafkaConsumer
import threading

LOG_FILE_PATH = 'log/message.log'
KAFKA_TOPIC = 'message'
KAFKA_BROKER = 'localhost:9092'

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def consume_messages():
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='log_debugger',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    for message in consumer:
        print(f"Consumed message: {message.value}")

consumer_thread = threading.Thread(target=consume_messages, daemon=True)
consumer_thread.start()

def process_log_file():
    with open(LOG_FILE_PATH, 'r') as file:
        file.seek(0, 2)
        # file.seek(0, 0)
        while True:
            line = file.readline()
            if line:
                if '{' in line and '}' in line:
                    try:
                        json_data = line[line.index('{'):line.rindex('}')+1] \
                            .replace('"', '\\\"').replace("{'", '{"').replace("': '", '": "').replace("'}", '"}').replace("', '", '", "')
                        json_object = json.loads(json_data)
                        producer.send(KAFKA_TOPIC, json_object)
                        print(f"Produced: {json_object}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON from line: {line}")
                        print(f"json data: {json_data}")
                        print(e)
            else:
                time.sleep(1)

if __name__ == "__main__":
    process_log_file()
