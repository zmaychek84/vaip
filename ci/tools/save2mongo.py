import json

from . import utility
from . import constant


def is_daily_regression(benchmark_data):
    user_name = benchmark_data.get("BUILD_USER", "")
    if user_name != "null":
        return False
    return True


def is_valid_benchmark(benchmark_data):
    keys = ("TEST_MODE", "MODEL_ZOO", "TARGET_TYPE")
    for key in keys:
        if not benchmark_data.get(key, ""):
            return False

    if benchmark_data["TEST_MODE"] == "performance":
        key_item = "Latency(ms)"
    elif benchmark_data["TEST_MODE"] in ("accuracy", "fast_accuracy"):
        key_item = "Accuracy"
    else:
        print("not support test mode!!!")
        return False
    for key, value in benchmark_data.items():
        if not isinstance(value, dict) or "Summary" not in value:
            continue
        if len(value["Summary"].get(key_item, {})) != 0:
            break
    else:
        return False

    for key, value in benchmark_data.items():
        if not isinstance(value, dict) or "Summary" not in value:
            continue
        if value["Summary"].get("OPS", 0):
            break
    else:
        return False

    return True


def save2mongo(benchmark_data_json):
    try:
        module_name = "pymongo"  # Replace with the name of the module you want to check
        utility.import_or_install(module_name)
        from pymongo import MongoClient
    except Exception as e:
        print("ERROR: import pymongo repo failed! %s" % e)
        return

    with open(benchmark_data_json) as file:
        benchmark_data = json.load(file)

    if not is_valid_benchmark(benchmark_data):
        print("ERROR: benchmark data is not valid.")
        return

    if is_daily_regression(benchmark_data):
        test_mode = benchmark_data["TEST_MODE"]
        test_mode = "accuracy" if test_mode == "fast_accuracy" else test_mode
        database_name = "daily-" + test_mode
        model_group = constant.MODELZOO_MAP.get(benchmark_data["MODEL_ZOO"], "")
        collention_name = f'{model_group}-{benchmark_data["TARGET_TYPE"]}'
        mongo_user_name = constant.MONGO_DAILY_USER
        mongo_user_pssword = constant.MONGO_DAILY_PASSWORD
    else:
        database_name = "development"
        collention_name = f'{benchmark_data["JOB_BASE_NAME"]}'
        mongo_user_name = constant.MONGO_TEST_USER
        mongo_user_pssword = constant.MONGO_TEST_PASSWORD

    print("database name: %s" % database_name)
    print("collection name: %s" % collention_name)
    try:
        uri = f"mongodb://{mongo_user_name}:{mongo_user_pssword}@{constant.MONGODB_HOST}:{constant.MONGODB_PORT}/"
        client = MongoClient(uri)
        db = client[database_name]
        collection = db[collention_name]

        # Insert data (use insert_many() for multiple records)
        x = collection.insert_one(benchmark_data)
        print(f"Inserted document ID: {x.inserted_id}")
    except Exception as e:
        print("ERROR: Inserted document FAILED! %s" % e)


if __name__ == "__main__":
    save2mongo("")
