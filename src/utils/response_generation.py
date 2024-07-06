from src.utils.stream_handler import StreamHandler


def generate_response(system_message, tokenizer, pipeline, logger, user_interaction, max_new_tokens=512):
    try:
        messages = [
            system_message,
            {"role": "user", "content": user_interaction},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        streamer = StreamHandler(tokenizer, skip_prompt=True)

        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

        pipeline(prompt, **generation_kwargs)

        return streamer.generated_text.strip()

    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
        return "I apologize, but I encountered an error while processing your request."
