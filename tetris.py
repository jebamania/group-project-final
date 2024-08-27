from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Initialize the emulator with the Tetris ROM
pyboy = PyBoy('Tetris.gb')
pyboy.set_emulation_speed(0)
assert pyboy.cartridge_title == "TETRIS"

# Start the Tetris game
tetris = pyboy.game_wrapper
tetris.start_game(timer_div=0x00)  # The timer_div works like a random seed in Tetris
pyboy.tick()  # To render screen after `.start_game`

# Start screen recording
pyboy.send_input(WindowEvent.SCREEN_RECORDING_TOGGLE)

# Keep playing the game until game over
while not tetris.game_over():
    pyboy.tick(1, True)
    pyboy.button("right")  # The playing "technique" is just to move the Tetromino to the right.

# Save the final screen
pyboy.screen.image.save("Tetris2.png")
pyboy.send_input(WindowEvent.SCREEN_RECORDING_TOGGLE)

# Output the final score
print("Game Over!")
print(f"Final Score: {tetris.score}")
print(f"Final Level: {tetris.level}")
print(f"Total Lines Cleared: {tetris.lines}")

# Stop the emulator
pyboy.stop()
