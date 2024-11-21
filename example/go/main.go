package main

import (
	"context"
	"os"

	sdk "github.com/traceloop/go-openllmetry/traceloop-sdk"
)

func main() {
	ctx := context.Background()

	traceloop, _ := sdk.NewClient(ctx, sdk.Config{
		APIKey: os.Getenv("TRACELOOP_API_KEY"),
	})
	defer func() { traceloop.Shutdown(ctx) }()
}
