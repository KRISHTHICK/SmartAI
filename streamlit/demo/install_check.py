# smart_installer.py
import subprocess

with open("requirements.txt") as f:
    packages = f.read().splitlines()

print("\n📦 Starting package installation one by one...\n")
for i, pkg in enumerate(packages, start=1):
    print(f"🧪 [{i}/{len(packages)}] Installing: {pkg}")
    try:
        subprocess.check_call(["pip", "install", pkg])
        print(f"✅ Success: {pkg}\n")
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {pkg}\n")

#Option:1
# step-1 -> pip install pipreqs
# step 2 -> then run above script or files with -> python install_check.py for this to generate requirements.txt

#----------------------------------------------------
# option 2
# Check After Install With pip list
# After your batch install, run:
# # Check installed packages
# pip list > installed.txt
# Then open installed.txt or search:
# # and display it in the app
# findstr streamlit installed.txt

#----------------------------------------------------

# option #4: Check if all packages are installed
# After running the above script, you can check if all required packages are installed.
# Then run this Python to find missing:
# # missing_check.py
# required = set(open("requirements.txt").read().splitlines())
# installed = set([line.split("==")[0] for line in open("current.txt").read().splitlines()])
# missing = required - installed
# print("📍 Missing packages:")
# for m in missing:
#     print(f"❌ {m}")

#--------------------------------------------------
# option #3: 
#Use pip install -r requirements.txt -v for Verbose Logging
# This will print detailed logs of what's happening:
# pip install -r requirements.txt -v

# Virutal enviriment Creation
# python -m venv venv
# venv\Scripts\activate  # Windows
# pip install -r requirements.txt
