/**
 * Relevance-Based Context Pruning - Production Grade
 * 
 * Prunes context to keep only messages relevant to current task.
 * Uses semantic similarity and task context for relevance scoring.
 */

import { roughTokenCountEstimation } from '../services/tokenEstimation.js'
import type { Message } from '../types/message.js'

export interface RelevanceScore {
  message: Message
  score: number
  reason: string
}

export interface PruningOptions {
  targetTokens: number
  taskContext?: string
  minRelevanceScore?: number
  preserveRecent?: number
  preserveTools?: boolean
  preserveErrors?: boolean
}

const STOP_WORDS = new Set([
  'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
  'her', 'was', 'one', 'our', 'out', 'has', 'have', 'they', 'will', 'would',
  'this', 'that', 'with', 'from', 'just', 'about', 'also', 'into', 'over',
])

/**
 * Extract significant keywords from text.
 */
function extractKeywords(text: string): Set<string> {
  const words = text.toLowerCase().split(/\s+/)
  const keywords = new Set<string>()

  for (const word of words) {
    const cleaned = word.replace(/[^a-z]/g, '')
    if (cleaned.length > 3 && !STOP_WORDS.has(cleaned)) {
      keywords.add(cleaned)
    }
  }

  return keywords
}

/**
 * Calculate keyword overlap between two texts.
 */
function calculateKeywordOverlap(text1: string, text2: string): number {
  const keywords1 = extractKeywords(text1)
  const keywords2 = extractKeywords(text2)

  let overlap = 0
  for (const keyword of keywords1) {
    if (keywords2.has(keyword)) {
      overlap++
    }
  }

  const total = keywords1.size + keywords2.size
  return total > 0 ? (2 * overlap) / total : 0
}

/**
 * Check if message contains tool calls.
 */
function hasToolCalls(message: Message): boolean {
  const content = typeof message.message?.content === 'string'
    ? message.message.content
    : ''

  if (typeof message.message?.content === 'string') {
    return content.includes('tool_use') || content.includes('function_call')
  }

  if (Array.isArray(message.message?.content)) {
    return message.message.content.some(block =>
      'type' in block && (block.type === 'tool_use' || block.type === 'tool_result')
    )
  }

  return false
}

/**
 * Check if message contains errors.
 */
function hasErrors(message: Message): boolean {
  const content = typeof message.message?.content === 'string'
    ? message.message.content
    : ''

  return content.includes('error') ||
         content.includes('fail') ||
         content.includes('exception') ||
         content.includes('Error')
}

/**
 * Calculate relevance score for a message.
 */
function calculateRelevance(
  message: Message,
  options: PruningOptions,
): RelevanceScore {
  const content = typeof message.message?.content === 'string'
    ? message.message.content
    : typeof message.message?.content === 'object' && message.message?.content !== null
      ? JSON.stringify(message.message.content)
      : ''

  const createdAt = message.message?.created_at ?? 0
  const now = Date.now()
  const ageHours = (now - createdAt) / (1000 * 60 * 60)

  let score = 0.5
  let reason = 'default'

  const keywordOverlap = options.taskContext
    ? calculateKeywordOverlap(content, options.taskContext)
    : 0

  score += keywordOverlap * 0.3

  if (keywordOverlap > 0.3) {
    reason = 'task_relevant'
    score += 0.2
  }

  if (hasToolCalls(message) && options.preserveTools) {
    reason = 'tool_call'
    score += 0.25
  }

  if (hasErrors(message) && options.preserveErrors) {
    reason = 'error_content'
    score += 0.3
  }

  if (ageHours < 1) {
    score += 0.15
    if (reason === 'default') reason = 'recent'
  } else if (ageHours < 24) {
    score += 0.1
  }

  if (message.message?.role === 'user') {
    score += 0.1
  }

  if (content.includes('important') || content.includes('critical')) {
    score += 0.2
    reason = 'important'
  }

  return {
    message,
    score: Math.min(1, score),
    reason,
  }
}

/**
 * Prune messages by relevance to task context.
 */
export function pruneByRelevance(
  messages: Message[],
  options: PruningOptions,
): Message[] {
  const targetTokens = options.targetTokens ?? 5000
  const minRelevanceScore = options.minRelevanceScore ?? 0.3
  const preserveRecent = options.preserveRecent ?? 3

  let totalTokens = 0
  const recentMessages = messages.slice(-preserveRecent)
  const olderMessages = messages.slice(0, -preserveRecent)

  const scored: RelevanceScore[] = olderMessages.map(msg => calculateRelevance(msg, options))

  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score
    const aTime = a.message.message?.created_at ?? 0
    const bTime = b.message.message?.created_at ?? 0
    return bTime - aTime
  })

  const result: Message[] = [...recentMessages]

  for (const { message, score } of scored) {
    if (score < minRelevanceScore) continue

    const content = typeof message.message?.content === 'string'
      ? message.message.content
      : typeof message.message?.content === 'object' && message.message?.content !== null
        ? JSON.stringify(message.message.content)
        : ''

    const tokens = roughTokenCountEstimation(content)

    if (totalTokens + tokens > targetTokens) {
      break
    }

    result.push(message)
    totalTokens += tokens
  }

  return result.sort((a, b) => (a.message?.created_at ?? 0) - (b.message?.created_at ?? 0))
}

/**
 * Get top N most relevant messages.
 */
export function getTopRelevantMessages(
  messages: Message[],
  options: PruningOptions,
  limit: number = 10,
): Message[] {
  const scored = messages.map(msg => calculateRelevance(msg, options))
  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, limit).map(s => s.message)
}

/**
 * Calculate relevance statistics.
 */
export function getRelevanceStats(
  messages: Message[],
  options: PruningOptions,
): {
  averageScore: number
  highRelevanceCount: number
  toolCallCount: number
  errorCount: number
} {
  const scored = messages.map(msg => calculateRelevance(msg, options))

  const averageScore = scored.length > 0
    ? scored.reduce((sum, s) => sum + s.score, 0) / scored.length
    : 0

  return {
    averageScore,
    highRelevanceCount: scored.filter(s => s.score > 0.7).length,
    toolCallCount: messages.filter(hasToolCalls).length,
    errorCount: messages.filter(hasErrors).length,
  }
}