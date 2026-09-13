"""python -m api.app.cli create-admin --email admin@example.com"""

import argparse
from getpass import getpass

from pydantic import EmailStr, TypeAdapter
from sqlalchemy import select

from .config import Settings
from .db import database
from .models import User
from .security import passwords


def main():
    parser = argparse.ArgumentParser(description="S10 administrative commands")
    parser.add_argument("command", choices=["create-admin"])
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default="Platform administrator")
    args = parser.parse_args()
    email = str(TypeAdapter(EmailStr).validate_python(args.email)).lower()
    password = getpass("Senha (12 a 128 caracteres): ")
    if not 12 <= len(password) <= 128 or password != getpass("Confirme a senha: "):
        parser.error("Senha invalida ou confirmacao diferente")
    engine, factory = database(Settings())
    try:
        with factory() as db:
            if db.scalar(select(User).where(User.email == email)):
                parser.error("E-mail ja cadastrado; nenhum usuario foi alterado")
            db.add(
                User(
                    name=args.name,
                    email=email,
                    password_hash=passwords.hash(password),
                    role="platform_admin",
                )
            )
            db.commit()
        print("Administrador criado.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
