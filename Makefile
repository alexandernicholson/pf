CC      ?= cc
ARCH    := $(shell uname -m)
ifneq (,$(filter $(ARCH),x86_64 amd64 i386 i686))
ARCHFLAGS ?= -march=native
else ifneq (,$(filter $(ARCH),arm64 aarch64))
ARCHFLAGS ?= -mcpu=native
else
ARCHFLAGS ?=
endif
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra
LDLIBS  ?= -lm

pf: pf.c unicode_tables.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(ARCHFLAGS) -pthread -o $@ pf.c $(LDFLAGS) $(LDLIBS)

unicode_tables.h: gen_unicode.py
	python3 gen_unicode.py $@

clean:
	rm -f pf

.PHONY: clean
