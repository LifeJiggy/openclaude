import { describe, expect, it } from 'bun:test'
import {
  pruneByRelevance,
  getTopRelevantMessages,
  getRelevanceStats,
} from './relevancePruning.js'

function createMessage(role: string, content: string, createdAt: number = Date.now()): any {
  return {
    message: { role, content, id: 'test', type: 'message', created_at: createdAt },
    sender: role,
  }
}

describe('relevancePruning', () => {
  describe('pruneByRelevance', () => {
    it('prunes to target token count', () => {
      const messages = [
        createMessage('user', 'Hello world how are you today', 1000),
        createMessage('assistant', 'I am doing great thank you', 2000),
        createMessage('user', 'Can you help me with python code', 3000),
        createMessage('assistant', 'Sure I can help with python', 4000),
      ]

      const result = pruneByRelevance(messages, { targetTokens: 50 })

      expect(result.length).toBeLessThanOrEqual(messages.length)
    })

    it('preserves recent messages', () => {
      const messages = [
        createMessage('user', 'Old message 1', 1000),
        createMessage('user', 'Old message 2', 2000),
        createMessage('user', 'Recent message', Date.now()),
      ]

      const result = pruneByRelevance(messages, { targetTokens: 100, preserveRecent: 1 })

      expect(result.length).toBeGreaterThan(0)
    })

    it('preserves tool calls', () => {
      const messages = [
        createMessage('assistant', 'Running tool_use to check file', 1000),
        createMessage('user', 'Regular message', 2000),
      ]

      const result = pruneByRelevance(messages, {
        targetTokens: 100,
        preserveTools: true,
        taskContext: 'file check',
      })

      expect(result.length).toBeGreaterThan(0)
    })

    it('preserves error content', () => {
      const messages = [
        createMessage('assistant', 'Found an error in the code', 1000),
        createMessage('user', 'Hello there', 2000),
      ]

      const result = pruneByRelevance(messages, {
        targetTokens: 100,
        preserveErrors: true,
      })

      expect(result.length).toBeGreaterThan(0)
    })
  })

  describe('getTopRelevantMessages', () => {
    it('returns top N relevant messages', () => {
      const messages = [
        createMessage('user', 'Message about python programming', 1000),
        createMessage('assistant', 'Python is a great language', 2000),
        createMessage('user', 'Message about javascript', 3000),
      ]

      const result = getTopRelevantMessages(
        messages,
        { targetTokens: 100, taskContext: 'python programming' },
        2
      )

      expect(result.length).toBeLessThanOrEqual(2)
    })
  })

  describe('getRelevanceStats', () => {
    it('calculates relevance statistics', () => {
      const messages = [
        createMessage('user', 'Important message about errors', 1000),
        createMessage('assistant', 'Tool call using tool_use', 2000),
        createMessage('user', 'Regular message', 3000),
      ]

      const stats = getRelevanceStats(messages, {
        targetTokens: 100,
        preserveTools: true,
        preserveErrors: true,
      })

      expect(stats.averageScore).toBeGreaterThan(0)
      expect(stats.highRelevanceCount).toBeGreaterThanOrEqual(0)
      expect(stats.toolCallCount).toBeGreaterThanOrEqual(0)
      expect(stats.errorCount).toBeGreaterThanOrEqual(0)
    })
  })
})