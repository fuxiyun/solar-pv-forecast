"""CLI entry point for the solar PV forecasting pipeline."""

import click
from loguru import logger

from solar_pv_forecast.utils import setup_logger, log_step


@click.group()
def main():
    """Solar PV Forecasting Pipeline for Germany."""
    setup_logger()


@main.command()
def run_all():
    """Run the full pipeline end-to-end."""
    from solar_pv_forecast.data.fetch_weather import main as fetch_weather
    from solar_pv_forecast.data.fetch_target import main as fetch_target
    from solar_pv_forecast.data.fetch_pv_capacity import main as fetch_pv
    from solar_pv_forecast.data.harmonise import main as harmonise
    from solar_pv_forecast.proxy.build_proxy import main as build_proxy
    from solar_pv_forecast.model.train import main as train
    from solar_pv_forecast.model.evaluate import main as evaluate

    with log_step("Full pipeline"):
        fetch_weather()
        fetch_target()
        fetch_pv()
        harmonise()
        build_proxy()
        train()
        evaluate()

    logger.info("Pipeline complete. See output/ for results.")


if __name__ == "__main__":
    main()
