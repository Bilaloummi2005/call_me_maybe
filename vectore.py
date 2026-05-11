
parameters = { 
      "a": {
        "type": "number"  
      },
      "b": {
        "type": "number"
      }
    }

for key, value in parameters.items():
    print(key)
    print(value.items())