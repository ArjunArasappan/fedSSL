# PyTorch Federated Self-Supervised Learning

*Need to Finish, talk about SSL, project implementation details, SimCLR, contrastive loss, etc. * 


## Environment Setup

Start by cloning the code example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/pytorch-federeated-self-supervised . && rm -rf flower && cd pytorch-federeated-self-supervised
```

This will create a new directory called `pytorch-federeated-self-supervised` containing the following files:

```
-- README.md           <- Your're reading this right now
-- main.py             <- Start federated simulation
-- client.py           <- Flower client constructor
-- model.py            <- Contains models and contrastive loss
-- utils.py            <- Utility functions (data loading, simulation settings)
-- test.py             <- Fine-tune and test pre-trained model
-- app.py              <- ServerApp/ClientApp for Flower-Next
-- conf/config.yaml    <- Configuration file
-- requirements.txt    <- Example dependencies
```


### Installing dependencies

Project dependencies are defined in `requirements.txt`. Install them with:

```shell
pip install -r requirements.txt
```



## Run with Flower Next (preview)

We conduct a 2-client setting to demonstrate how to run federated LLM fine-tuning with Flower Next.
Please follow the steps below:

1. Start the long-running Flower server (SuperLink)
   ```bash
   flower-superlink --insecure
   ```
2. Start the long-running Flower client (SuperNode)
   ```bash
   # In a new terminal window, start the first long-running Flower client:
   flower-client-app app:client1 --insecure
   ```
   ```bash
   # In another new terminal window, start the second long-running Flower client:
   flower-client-app app:client2 --insecure
   ```
3. Run the Flower App
   ```bash
   # With both the long-running server (SuperLink) and two clients (SuperNode) up and running,
   # we can now run the actual Flower App:
   flower-server-app app:server --insecure
   ```


## Expected Results

Add results. 