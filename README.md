## Dependencies
You may only install the following libraries mannually with Conda to run PermLLM

*   Basic libraries
    ```
    pytorch (newest version is ok I guess)
    ```

*   Required by ChatGLM-6B
    ```
    transformers=4.27.1  (APIs change often, so other versions doesn't work!)
    sentencepiece
    ```
    To check, run `llm_bases/chatglm6b.py`

*   Required by PermLLM
    ```
    tenseal  (For BFV homomorphic encryption)
    ```
