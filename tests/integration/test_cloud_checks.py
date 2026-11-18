from sales_poc.checks import check_s3


def test_s3_write_read_delete():
    assert "wrote, read and deleted" in check_s3()
