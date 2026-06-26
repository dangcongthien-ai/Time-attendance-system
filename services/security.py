import base64
import hashlib
import hmac
import os
import secrets
import time


class PasswordHasher:
    prefix = "pbkdf2_sha256"
    default_iterations = 260_000

    @classmethod
    def hash(cls, password: str) -> str:
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("ascii"),
            cls.default_iterations,
        )
        encoded = base64.b64encode(digest).decode("ascii")
        return f"{cls.prefix}${cls.default_iterations}${salt}${encoded}"

    @classmethod
    def verify(cls, password: str, stored: str | None) -> bool:
        if not stored:
            return False

        if not stored.startswith(f"{cls.prefix}$"):
            return hmac.compare_digest(password, stored)

        try:
            _, iterations, salt, encoded = stored.split("$", 3)
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("ascii"),
                int(iterations),
            )
            expected = base64.b64encode(digest).decode("ascii")
            return hmac.compare_digest(expected, encoded)
        except Exception:
            return False

    @classmethod
    def needs_rehash(cls, stored: str | None) -> bool:
        return not stored or not stored.startswith(f"{cls.prefix}$")


class AdminSessionManager:
    def __init__(self, secret_key: str | None = None, max_age_seconds: int = 900):
        self.secret_key = (secret_key or os.environ.get("APP_SECRET_KEY") or "dev-secret-change-me").encode("utf-8")
        self.max_age_seconds = max_age_seconds

    def create_token(self, now: float | None = None) -> str:
        issued_at = str(int(now or time.time()))
        signature = self._sign(issued_at)
        return f"{issued_at}.{signature}"

    def verify_token(self, token: str | None, now: float | None = None) -> bool:
        if not token or "." not in token:
            return False

        issued_at, signature = token.split(".", 1)
        if not issued_at.isdigit():
            return False

        expected = self._sign(issued_at)
        if not hmac.compare_digest(signature, expected):
            return False

        age = int(now or time.time()) - int(issued_at)
        return 0 <= age <= self.max_age_seconds

    def _sign(self, issued_at: str) -> str:
        return hmac.new(self.secret_key, issued_at.encode("ascii"), hashlib.sha256).hexdigest()
