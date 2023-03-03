from asciimatics.screen import Screen
from time import sleep
import numpy as np

X = 74
Y = 33
COORD = np.array([X,Y])
COLOUR_GREEN = 2

barriers = [(20, x) for x in range(20)]

"""
COLOUR_BLACK = 0
COLOUR_RED = 1
COLOUR_GREEN = 2
COLOUR_YELLOW = 3
COLOUR_BLUE = 4
COLOUR_MAGENTA = 5
COLOUR_CYAN = 6
COLOUR_WHITE = 7

A_BOLD = 1
A_NORMAL = 2
A_REVERSE = 3
A_UNDERLINE = 4

{screen.dimensions}


"""


def demo(screen):
	x = screen.dimensions[1]
	y = screen.dimensions[0]

	leftright = int(x/4)
	updown = int(y/4)
	for i in range(updown, y-updown):
		screen.print_at(f'▍', leftright, i, COLOUR_GREEN)
		screen.print_at(f'▍', x-leftright, i, COLOUR_GREEN)


	for j in range(leftright, x-leftright):
		screen.print_at(f'▁', j, 0, COLOUR_GREEN)
		screen.print_at(f'▁', j, y-1, COLOUR_GREEN)

	for i in barriers:
		screen.print_at(f'▍', i[0]+(leftright+1), i[1]+1, 3)


	while True:
		for i in range(10):
			# screen.print_at(f'{i} {leftright} {x-leftright}', 10, 11, COLOUR_GREEN)
			screen.print_at(f'h', 74+(leftright+1), 33+1, COLOUR_GREEN)
			screen.refresh()
			sleep(2)


Screen.wrapper(demo)