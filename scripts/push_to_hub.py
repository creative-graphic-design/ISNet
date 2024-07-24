import logging

from isnet import ISNetConfig, ISNetImageProcessor, convert_from_checkpoint

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def push_isnet_to_hub(
    push_to_repo_name: str,
    checkpoint_filename: str,
    checkpoint_repo_name: str = "creative-graphic-design/ISNet-checkpoints",
) -> None:
    config = ISNetConfig()
    model = convert_from_checkpoint(
        repo_id=checkpoint_repo_name,
        filename=checkpoint_filename,
        config=config,
    )
    processor = ISNetImageProcessor()

    config.register_for_auto_class()
    model.register_for_auto_class()
    processor.register_for_auto_class()

    logger.info(f"Push model to the hub: {push_to_repo_name}")
    model.push_to_hub(push_to_repo_name, private=True)

    logger.info(f"Push processor to the hub: {push_to_repo_name}")
    processor.push_to_hub(push_to_repo_name, private=True)


def main():
    push_isnet_to_hub(
        push_to_repo_name="creative-graphic-design/ISNet-general-use",
        checkpoint_filename="isnet-general-use.pth",
    )


if __name__ == "__main__":
    main()
