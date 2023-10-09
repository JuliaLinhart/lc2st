from setuptools import setup, find_packages

setup(
    name="valdiags",
    packages=find_packages(include=["valdiags", "valdiags.*", "hnpe", "hnpe.*"]),
    install_requires=["sbi", "sbibm", "lampe", "zuko", "tueplots", "seaborn"],
)
