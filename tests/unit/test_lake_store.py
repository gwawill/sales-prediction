import pytest
from botocore.exceptions import ClientError

from sales_poc.lake.store import LocalStore, S3Store


def test_local_conditional_write(tmp_path):
    store = LocalStore(tmp_path)

    assert store.get_with_tag("a/b.txt") == (None, None)
    assert store.put_if_unchanged("a/b.txt", b"one", None)          # missing, as expected
    assert not store.put_if_unchanged("a/b.txt", b"two", None)      # now it exists

    data, tag = store.get_with_tag("a/b.txt")
    assert data == b"one"
    store.put("a/b.txt", b"changed by someone else")
    assert not store.put_if_unchanged("a/b.txt", b"two", tag)       # stale version
    assert store.get("a/b.txt") == b"changed by someone else"

    _, tag = store.get_with_tag("a/b.txt")
    assert store.put_if_unchanged("a/b.txt", b"two", tag)           # current version
    assert store.get("a/b.txt") == b"two"


def client_error(status, code):
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "PutObject")


class FakeS3:
    def __init__(self, fail=None):
        self.fail, self.calls = fail, []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail


def test_s3_conditional_write_sends_the_right_condition():
    client = FakeS3()
    store = S3Store(client, "bucket", prefix="lake")

    assert store.put_if_unchanged("k", b"x", '"abc"')
    assert store.put_if_unchanged("k", b"x", None)

    assert client.calls[0]["IfMatch"] == '"abc"' and "IfNoneMatch" not in client.calls[0]
    assert client.calls[1]["IfNoneMatch"] == "*" and "IfMatch" not in client.calls[1]
    assert client.calls[0]["Key"] == "lake/k"


@pytest.mark.parametrize(("status", "code"), [(412, "PreconditionFailed"), (409, "ConditionalRequestConflict")])
def test_s3_says_the_file_changed(status, code):
    store = S3Store(FakeS3(fail=client_error(status, code)), "bucket")
    assert store.put_if_unchanged("k", b"x", '"abc"') is False


def test_other_s3_errors_are_not_hidden():
    store = S3Store(FakeS3(fail=client_error(403, "AccessDenied")), "bucket")
    with pytest.raises(ClientError):
        store.put_if_unchanged("k", b"x", '"abc"')
