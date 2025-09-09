import os, wandb

# paste your key here if not already in WANDB_API_KEY
os.environ["WANDB_API_KEY"] = "0cf6db49c1856f2fefe44fb271e2b715e0cf4f91"

api = wandb.Api()
me = api.viewer   # not api.viewer()

print("Username:", me.username)
print("Email:", me.email)