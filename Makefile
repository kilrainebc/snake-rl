.PHONY: install train demo_untrained demo_trained plot gifs visuals clean all

# Install all dependencies
install:
	pip3 install -r requirements.txt

# Train the agent (1M timesteps, ~25 min on CPU)
train:
	python3 train.py

# Watch a random (untrained) agent play
demo_untrained:
	python3 demo.py --random --episodes 5

# Watch the trained agent play
demo_trained:
	python3 demo.py --episodes 5

# Generate learning curve plots
plot:
	MPLBACKEND=Agg python3 plot_curves.py

# Record demo GIFs (backup for presentations)
gifs:
	python3 record_gif.py --episodes 5

# Generate presentation visuals (observation heatmap, game screenshots)
visuals:
	MPLBACKEND=Agg python3 generate_visuals.py

# Remove generated artifacts (keeps code, removes outputs)
clean:
	rm -f output/snake_ppo.zip output/training_metrics.json
	rm -f output/learning_curves.png
	rm -f output/demo_*.gif
	rm -f output/observation_channels.png output/game_screenshot*.png

# Full pipeline from scratch
all: install train plot gifs visuals
