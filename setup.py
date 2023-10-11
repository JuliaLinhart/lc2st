from setuptools import setup, find_packages

setup(
    name="lc2st",
    packages=find_packages(include=["lc2st", "lc2st.*", "hnpe", "hnpe.*"]),
    install_requires=["sbi", "sbibm", "lampe", "zuko", "tueplots", "seaborn"],
)
