import { defineConfig } from "vite";
export default defineConfig({ base: "./", build: { target: "esnext" }, server: { port: 5173 }, test: { include: ["src/**/*.test.ts"] } } as any);
