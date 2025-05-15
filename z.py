import hashlib, pyotp

shared_secret = "mohammedzayantariq12@gmail.comHENNGECHALLENGE004"
totp = pyotp.TOTP(
    shared_secret,
    digits=10,
    digest=hashlib.sha512,
    interval=30
)
code = totp.now()
print(code)
