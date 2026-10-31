"""Hugging Face Spaces entry point for Koumba."""

from koumba.app import CSS, build_app


def main() -> None:
    """Start the demo on the port Gradio gives to Spaces."""

    # Spaces sends public traffic to 0.0.0.0:7860.  Leaving port unset keeps
    # that default, and share=False means no temporary public Gradio tunnel.
    build_app().launch(
        server_name="0.0.0.0",
        share=False,
        css=CSS,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
