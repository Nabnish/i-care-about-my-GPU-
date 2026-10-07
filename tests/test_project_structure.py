from importlib import import_module


def test_project_modules_are_importable():
    modules = (
        "simulator.gpu",
        "simulator.faults",
        "simulator.telemetry",
        "simulator.monitoring",
        "preprocessing.cleaning",
        "preprocessing.windows",
        "preprocessing.features",
        "models.risk_model",
        "models.observability",
        "models.calibration",
        "alerts.alert_logic",
        "evaluation.metrics",
        "evaluation.calibration",
        "evaluation.experiments",
    )

    for module in modules:
        import_module(module)
