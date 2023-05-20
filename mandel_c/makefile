CC = gcc
CFLAGS = -O3
DEBUG_FLAGS = -g -Wall -Wextra -fsanitize=address,undefined

mandel: main.c bitmap.h
	$(CC) $(CFLAGS) -mavx -lm -o mandel main.c

debug: main.c bitmap.h
	$(CC) $(DEBUG_FLAGS) -mavx -lm -o mandel main.c

test: mandel
	./mandel out.bmp
	feh out.bmp

clean:
	rm mandel *.bmp

.PHONY: clean test debug