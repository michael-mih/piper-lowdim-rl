from setuptools import find_namespace_packages, setup


setup(
    name="piper-force-rl",
    version="0.1.0",
    description="Piper force RL controllers and MuJoCo training scripts",
    package_dir={"": "src"},
    packages=find_namespace_packages(
        where="src",
        include=["controllers*", "fsm*", "learning*", "scripts*"],
    ),
    python_requires=">=3.8",
    extras_require={
        "ppo": ["torch"],
    },
    entry_points={
        "console_scripts": [
            "train-policy=scripts.train_policy:main",
            "train-ppo-grasp=scripts.train_ppo_grasp:main",
            "run-ppo-grasp=scripts.run_ppo_grasp:main",
        ],
    },
)
