__copyright__ = """\
Copyright (C) m-click.aero GmbH

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

import logging
from datetime import datetime
from pathlib import Path

from cryptography.hazmat.primitives.serialization import pkcs12
from OpenSSL import crypto
from requests.adapters import HTTPAdapter
from urllib3.contrib.pyopenssl import PyOpenSSLContext

try:
    from ssl import PROTOCOL_TLS as ssl_protocol
except ImportError:
    from ssl import PROTOCOL_SSLv23 as ssl_protocol


def check_cert_not_after(certificate):
    if certificate.not_valid_after < datetime.utcnow():
        raise ValueError(
            "Client certificate expired: Not After: "
            f"{certificate.not_valid_after:%Y-%m-%d %H:%M:%SZ}"
        )


def create_pyopenssl_sslcontext(data: bytes, password: bytes):
    private_key, certificate, additional = pkcs12.load_key_and_certificates(
        data, password
    )
    assert certificate is not None
    check_cert_not_after(certificate)
    ssl_context = PyOpenSSLContext(ssl_protocol)
    ssl_context._ctx.use_certificate(crypto.X509.from_cryptography(certificate))
    logging.info(
        f"Certificate found with subject {certificate.subject}, "
        f"expire on {certificate.not_valid_after}"
    )
    if additional:
        for ca_cert in additional:
            check_cert_not_after(ca_cert)
            ssl_context._ctx.add_extra_chain_cert(
                crypto.X509.from_cryptography(ca_cert)
            )
    ssl_context._ctx.use_privatekey(
        crypto.PKey.from_cryptography_key(private_key)
    )
    return ssl_context


class Pkcs12Adapter(HTTPAdapter):
    def __init__(self, filename: Path, password: str):
        pkcs12_data = filename.read_bytes()
        if isinstance(password, bytes):
            pkcs12_password_bytes = password
        else:
            pkcs12_password_bytes = password.encode("utf8")
        self.ssl_context = create_pyopenssl_sslcontext(
            pkcs12_data, pkcs12_password_bytes
        )
        super().__init__()

    def init_poolmanager(self, *args, **kwargs):
        if self.ssl_context:
            kwargs["ssl_context"] = self.ssl_context
        return super(Pkcs12Adapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        if self.ssl_context:
            kwargs["ssl_context"] = self.ssl_context
        return super(Pkcs12Adapter, self).proxy_manager_for(*args, **kwargs)
