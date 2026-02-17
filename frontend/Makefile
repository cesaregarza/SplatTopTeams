SHELL := /bin/bash

NPM ?= npm

.PHONY: help install dev build preview clean

help:
	@echo "Frontend targets:"
	@echo "  make install   Install npm dependencies"
	@echo "  make dev       Run Vite dev server"
	@echo "  make build     Build production bundle"
	@echo "  make preview   Preview production bundle"
	@echo "  make clean     Remove node_modules and dist"

install:
	$(NPM) install

dev: install
	$(NPM) run dev

build: install
	$(NPM) run build

preview: install
	$(NPM) run preview

clean:
	rm -rf node_modules dist
