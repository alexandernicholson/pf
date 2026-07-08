CC      ?= clang
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -mcpu=native
LDFLAGS ?= -lm

pf: pf.c unicode_tables.h
	$(CC) $(CFLAGS) -o $@ pf.c $(LDFLAGS)

unicode_tables.h: gen_unicode.py
	python3 gen_unicode.py $@

clean:
	rm -f pf

.PHONY: clean
