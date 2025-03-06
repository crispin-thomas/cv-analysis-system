import json
from logging import getLogger

logger = getLogger(__name__)


class QueryEngine:
    def __init__(self, llm_service, cv_database):
        self.llm_service = llm_service
        self.cv_database = cv_database
        self.conversation_context = {}

    def process_query(self, query, user_id=None, conversation_id=None):
        """Process a natural language query about CVs"""
        # Track conversation context
        context_key = (
            f"{user_id}_{conversation_id}" if user_id and conversation_id else None
        )
        context = self._get_conversation_context(context_key)

        # Get relevant CV data
        cv_data = self.cv_database.get_all_cvs()

        # Create prompt
        prompt = self._create_query_prompt(query, cv_data, context)

        try:
            # Get response from LLM
            response = self.llm_service.query(prompt)

            # Update conversation context
            if context_key:
                self._update_conversation_context(context_key, query, response)

            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I'm sorry, I encountered an error processing your query: {str(e)}"

    def _create_query_prompt(self, query, cv_data, context):
        """Create a prompt for the LLM to answer questions about CVs"""
        cv_json = json.dumps(cv_data, indent=2)
        context_str = (
            json.dumps(context, indent=2) if context else "No previous context."
        )

        prompt = f"""
        You are an expert CV analysis assistant. Answer the following question about these CVs.
        
        CV DATA:
        {cv_json}
        
        PREVIOUS CONVERSATION CONTEXT:
        {context_str}
        
        USER QUESTION:
        {query}
        
        Provide a clear and concise answer based on the CV data provided. 
        If you're comparing candidates, highlight the key differences.
        If you're listing candidates with specific qualifications, be precise about why they match.
        If you need more information, specify what additional details would help.
        """
        return prompt

    def _get_conversation_context(self, context_key):
        """Get the conversation context for a specific user/conversation"""
        if not context_key:
            return None

        if context_key not in self.conversation_context:
            self.conversation_context[context_key] = []

        return self.conversation_context[context_key]

    def _update_conversation_context(self, context_key, query, response):
        """Update the conversation context with the new query and response"""
        if not context_key:
            return

        context = self._get_conversation_context(context_key)
        context.append({"query": query, "response": response})

        # Keep only the last 5 exchanges for context
        self.conversation_context[context_key] = context[-5:]
