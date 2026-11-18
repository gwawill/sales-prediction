from sales_poc.checks import check_kafka, check_s3, check_s3_vectors


def test_s3_write_read_delete():
    assert "wrote, read and deleted" in check_s3()


def test_s3_vectors_store_and_search():
    assert "stored and found one vector" in check_s3_vectors()


def test_kafka_send_and_read_back():
    assert "sent and read back one message" in check_kafka()
